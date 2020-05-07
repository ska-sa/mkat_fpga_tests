# import threading
import base64
import glob
import logging
import operator
import os
import pwd
import Queue
import random
import re
import signal
import socket
import struct
import subprocess
import time
import warnings
import math
from ast import literal_eval as evaluate
from collections import Mapping, OrderedDict
from contextlib import contextmanager
from inspect import getframeinfo, stack
from socket import inet_ntoa
from struct import pack
from datetime import datetime

import h5py
import katcp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from casperfpga.utils import threaded_create_fpgas_from_hosts
from corr2.data_stream import StreamAddress
from Crypto.Cipher import AES
from mkat_fpga_tests.aqf_utils import *
from nose.plugins.attrib import attr
# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
# from memory_profiler import profile as DetectMemLeaks
from nosekatreport import Aqf

from Logger import LoggingClass

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

# I'm sure there's a better way///
_logger = LoggingClass()
LOGGER = _logger.logger

# Max range of the integers coming out of VACC
VACC_FULL_RANGE = float(2 ** 31)

# Katcp default timeout
cam_timeout = 180


__all__ = ["all_nonzero_baselines", "AqfReporter", "baseline_checker", "complexise", "CSV_Reader",
    "decode_passwd", "disable_warnings_messages", "encode_passwd", "executed_by",
    "flatten", "FPGA_Connect", "get_bit_flag", "get_delay_bounds", "get_dsim_source_info",
    "get_pfb_counts", "get_set_bits", "get_vacc_offset", "GetSensors", "human_readable_ip",
    "ignored", "init_dsim_sources", "int2ip", "ip2int", "iterate_recursive_dict", "loggerise",
    "magnetise", "nonzero_baselines", "normalise", "normalised_magnitude", "Report_Images",
    "RetryError", "retryloop", "RunTestWithTimeout", "TestTimeout", "UtilsClass", "wipd",
    "array_release_x", "subset", "beamforming", "zero_baselines"]


class RetryError(Exception):
    pass


class UtilsClass(object):

    def get_real_clean_dump(self, discard=0, quiet=False):
        """
            The data queue is cleared by calling get_clean_dump repeatedly
            until the procedure actually takes as long as an integration.
        """
        time_diff = 0
        retries = 20
        while time_diff < 0.1:
            try:
                start_time = time.time()
                data = self.receiver.get_clean_dump(discard=discard)
                self.assertIsInstance(data, dict)
                time_diff = time.time() - start_time
                self.logger.info("Time difference between capturing dumps = {}".format(time_diff))
                retries -= 1
                if retries == 0:
                    raise Exception('Could not retrieve clean SPEAD accumulation, retries exausted.')
            except AssertionError:
                errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
                if not quiet:
                    self.Error(errmsg, exc_info=True)
                return False
            except Exception as e:
                if not quiet:
                    self.Error(e, exc_info=True)
                return False
        return data

    def clear_all_delays(self):
        """Clears all delays on all fhosts.
            Param: object
            Return: Boolean
        """
        try:
            no_fengines = self.cam_sensors.get_value("n_fengs")
            int_time = self.cam_sensors.get_value("int_time")
            self.logger.info("Retrieving test parameters via CAM Interface")
        except Exception:
            no_fengines = len(self.correlator.fops.fengines)
            self.Error("Retrieving number of fengines via corr object: %s" % no_fengines, exc_info=True)

        delay_coefficients = ["0,0:0,0"] * no_fengines
        try:
            # TODO Not sure why this was done?
            #dump = self.receiver.get_clean_dump(discard=0)
            #deng_timestamp = self.dhost.registers.sys_clkcounter.read().get("timestamp")
            #discard = 0
            #while True:
            #    dump = self.receiver.data_queue.get(timeout=10)
            #    dump_timestamp = dump["dump_timestamp"]
            #    time_diff = np.abs(dump_timestamp - deng_timestamp)
            #    self.logger.info("Time difference between dump_timestamp and deng_timestamp = {}".format(time_diff))
            #    if time_diff < 1:
            #        break
            #    if discard > 10:
            #        raise AssertionError
            #    discard += 1
            #errmsg = "Dump timestamp (%s) is not in-sync with epoch (%s) [diff: %s]" % (
            #    dump_timestamp,
            #    deng_timestamp,
            #    time_diff,
            #)

            # Just set the time from cmc time, if CMCs are synchronised there is no need to to get time from dump 
            #num_int = int(self.conf_file["instrument_params"]["num_int_delay_load"])
            #t_apply = dump_timestamp + (num_int * int_time)
            buffer_time = 1
            delay_load_lead_time = float(self.conf_file['instrument_params']['delay_load_lead_time']) + buffer_time
            curr_time = time.time()
            curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
            t_apply = curr_time + delay_load_lead_time
            t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
            self.Step("Current cmc time is {}, delays will be cleared at {}".format(
                      curr_time_readable,
                      t_apply_readable)
            )
            reply, informs = self.corr_fix.katcp_rct.req.delays(self.corr_fix.feng_product_name,
                t_apply, *delay_coefficients, timeout = 30)
            errmsg = "Delays command could not be executed: {}".format(reply)
            assert reply.reply_ok(), errmsg
            # This is the old method for reading back delays
            #t_apply_dumps = math.ceil(delay_load_lead_time / int_time)
            # end_time = 0
            # while True:
            #    _give_up -= 1
            #    try:
            #        LOGGER.info('Waiting for the delays to be updated: %s retry' % _give_up)
            #        reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
            #        assert reply.reply_ok()
            #    except Exception:
            #        LOGGER.exception("Weirdly I couldn't get the sensor values, fix it and figure it out")
            #    else:
            #        delays_updated = list(set([int(i.arguments[-1]) for i in informs
            #                                    if '.cd.delay' in i.arguments[2]]))[0]
            #        if delays_updated:
            #            LOGGER.info('Delays have been successfully set')
            #            end_time = time.time()
            #            break
            #    if _give_up == 0:
            #        LOGGER.error("Could not confirm the delays in the time stipulated, exiting")
            #        break
            # time_end = abs(end_time - start_time)
            #dump = self.receiver.get_clean_dump(discard=(t_apply_dumps + 2))
            #_max = int(np.max(np.angle(dump["xeng_raw"][:, 33, :][5:-5])))
            #_min = int(np.min(np.angle(dump["xeng_raw"][:, 0, :][5:-5])))
            #errmsg = "Max/Min delays found: %s/%s ie not cleared" % (_max, _min)
            #assert _min == _max == 0, errmsg
            #self.logger.info(
            #    "Delays cleared successfully. Dump timestamp is in-sync with epoch: {}".format(
            #        time_diff
            #    )
            #)
            timeout = delay_load_lead_time * 2
            start_time = time.time()
            while True:
                if self._confirm_delays(delay_coefficients, err_margin = 0):
                    while (t_apply - curr_time) > 0:
                        # Keep the que clean while waiting
                        dump = self.receiver.get_clean_dump()
                        curr_time = time.time()
                    break
                elif (time.time() - start_time) > timeout:
                    errmsg = "Delays were not cleared."
                    raise AssertionError
            return True
        except AssertionError:
            self.logger.warning(errmsg)
        except TypeError:
            self.logger.exception("Object has no attributes")
            return False
        except Exception as e:
            self.logger.exception("Exception occured during clearing of delays: {}".format(e))
        return False

    def start_katsdpingest_docker(
        self,
        beam_ip,
        beam_port,
        partitions,
        channels=4096,
        ticks_between_spectra=8192,
        channels_per_heap=256,
        spectra_per_heap=256,
    ):
        """ Starts a katsdpingest docker containter. Kills any currently running instances.

        Returns
        -------
            False if katsdpingest docer not started
            True if katsdpingest docker started
        """
        user_id = pwd.getpwuid(os.getuid()).pw_uid
        cmd = [
            "docker",
            "run",
            "-u",
            "{}".format(user_id),
            "-d",
            "--net=host",
            "-v",
            "/ramdisk:/ramdisk",
            "sdp-docker-registry.kat.ac.za:5000/katsdpingest:cbf_testing",
            "bf_ingest.py",
            "--cbf-spead={}+{}:{} ".format(beam_ip, partitions - 1, beam_port),
            "--channels={}".format(channels),
            "--ticks-between-spectra={}".format(ticks_between_spectra),
            "--channels-per-heap={}".format(channels_per_heap),
            "--spectra-per-heap={}".format(spectra_per_heap),
            "--file-base=/ramdisk",
            "--log-level=DEBUG",
        ]

        self.stop_katsdpingest_docker()
        try:
            LOGGER.info("Executing docker command to run KAT SDP injest node")
            output = subprocess.check_output(cmd)
        except subprocess.CalledProcessError:
            errmsg = (
                "Could not start sdp-docker-registry container, "
                "ensure SDP Ingest image has been built successfully"
            )
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        time.sleep(5)
        try:
            output = subprocess.check_output(["/usr/bin/docker", "ps"])
        except subprocess.CalledProcessError:
            return False
        output = output.split()
        sdp_instance = [idx for idx, s in enumerate(output) if "sdp-docker-registry.kat.ac.za" in s]
        # If sdp-docker-registry not found it is not running, return false

        return True if sdp_instance else False


    @staticmethod
    def stop_katsdpingest_docker():
        """ Finds if a katsdpingest docker containter is running and kills it.

        Returns
        -------
            False if katsdpingest docker container not found or not running
            True if katsdpingest docker container found and stopped
        """
        try:
            output = subprocess.check_output(["/usr/bin/docker", "ps"])
        except subprocess.CalledProcessError:
            return False
        output = output.split()
        sdp_instance = [idx for idx, s in enumerate(output) if "sdp-docker-registry.kat.ac.za" in s]
        # If sdp-docker-registry not found it is not running, return false
        # Kill all instances found
        if sdp_instance:
            for idx in sdp_instance:
                try:
                    kill_output = subprocess.check_output(["docker", "kill", output[idx - 1]])
                except subprocess.CalledProcessError:
                    errmsg = "Could not kill sdp-docker-registry container"
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                    return False
                killed_id = kill_output.split()[0]
                if killed_id != output[idx - 1]:
                    errmsg = "Could not kill sdp-docker-registry container"
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                    return False

        else:
            return False
        return True


    def capture_beam_data(
        self,
        beam,
        beam_dict=None,
        ingest_kcp_client=None,
        capture_time=0.4,
        start_only=False,
        stop_only=False,
    ):
        """ Capture beamformer data

        Parameters
        ----------
        beam (beam_0x, beam_0y):
            Polarisation to capture beam data
        beam_dict:
            Dictionary containing input:weight key pairs e.g.
            beam_dict = {'m000_x': 1.0, 'm000_y': 1.0}
            If beam_dict = None weights will not be set
        ingest_kcp_client:
            katcp client for ingest node, if None one will be created.
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
            start_only:
                Only start a capture and return, capture_time will be ignored. Only returns a ingest_kcp_client handle
            stop_only:
                Only stop a capture and return data, this will fail if a capture was not started. Requires a ingest_kcp_client.

        """
        beamdata_dir = "/ramdisk"
        _timeout = 60

        # Create a katcp client to connect to katcpingest if one not specified
        if ingest_kcp_client is None:
            if os.uname()[1] == "cmc2":
                ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc2"]
            elif os.uname()[1] == "cmc3":
                ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc3"]
            else:
                ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node"]
            ingst_nd_p = self.corr_fix._test_config_file["beamformer"]["ingest_node_port"]
            try:
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
                errmsg = "Failed to execute katcp request."
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)

        # Build new dictionary with only the requested beam keys:value pairs
        # Put into an ordered list
        in_wgts = {}
        beam_pol = beam[-1]
        if beam_dict:
            for key in beam_dict:
                if key.find(beam_pol) != -1:
                    in_wgts[key] = beam_dict[key]

            n_ants = self.cam_sensors.get_value("n_ants")
            if len(in_wgts) != n_ants:
                errmsg = "Number of weights in beam_dict does not equal number of antennas. Programmatic error."
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
            weight_list = []
            for key in sorted(in_wgts.iterkeys()):
                weight_list.append(in_wgts[key])
            Aqf.step("Weights to set: {}".format(weight_list))

            Aqf.step(
                "Setting input weights, this may take a long time, check log output for progress..."
            )
            try:
                reply, informs = self.corr_fix.katcp_rct.req.beam_weights(
                    beam, *weight_list, timeout=60
                )
                assert reply.reply_ok()
            except AssertionError:
                Aqf.failed("Beam weights not successfully set: {}".format(reply))
            except Exception:
                errmsg = "Failed to execute katcp requests"
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                LOGGER.info("{} weights set to {}".format(beam, reply.arguments[1:]))
                Aqf.passed("Antenna input weights for {} set to: {}".format(beam, reply.arguments[1:]))

            # Old method of setting weights
            # print_list = ''
            # for key in in_wgts:
            #    LOGGER.info('Confirm that antenna input ({}) weight has been set to the desired weight.'.format(
            #        key))
            #    try:
            #        reply, informs = self.corr_fix.katcp_rct.req.beam_weights(
            #            beam, key, in_wgts[key])
            #        assert reply.reply_ok()
            #    except AssertionError:
            #        Aqf.failed(
            #            'Beam weights not successfully set: {}'.format(reply))
            #    else:
            #        LOGGER.info('Antenna input {} weight set to {}'.format(
            #            key, reply.arguments[1]))
            #        print_list += ('{}:{}, '.format(key, reply.arguments[1]))
            #        in_wgts[key] = float(reply.arguments[1])
            # Aqf.passed('Antenna input weights set to: {}'.format(print_list[:-2]))

        if not (stop_only):
            try:
                LOGGER.info("Issue {} capture start via CAM int".format(beam))
                for i in xrange(2):
                    reply, informs = self.corr_fix.katcp_rct.req.capture_meta(beam, timeout=60)
                errmsg = "Failed to issue new Metadata: {}".format(str(reply))
                assert reply.reply_ok(), errmsg
                reply, informs = self.corr_fix.katcp_rct.req.capture_start(beam, timeout=60)
                errmsg = "Failed to issue capture_start for beam {}: {}".format(beam, str(reply))
                assert reply.reply_ok(), errmsg
            except AssertionError:
                errmsg = " .".join([errmsg, "Failed to start Data transmission."])
                Aqf.failed(errmsg)
            try:
                LOGGER.info("Issue ingest node capture-init.")
                reply, informs = ingest_kcp_client.blocking_request(
                    katcp.Message.request("capture-init"), timeout=_timeout
                )
                errmsg = "Failed to issues ingest node capture-init: {}".format(str(reply))
                assert reply.reply_ok(), errmsg
            except Exception:
                errmsg = "Failed to execute katcp request"
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            if start_only:
                LOGGER.info("Only beam capture start issued.")
                return ingest_kcp_client
            LOGGER.info("Capturing beam data for {} seconds".format(capture_time))
            time.sleep(capture_time)
        try:
            LOGGER.info("Issue ingest node capture-done.")
            reply, informs = ingest_kcp_client.blocking_request(
                katcp.Message.request("capture-done"), timeout=_timeout
            )
            errmsg = "Failed to issues ingest node capture-done: {}".format(str(reply))
            assert reply.reply_ok(), errmsg
        except Exception:
            Aqf.failed(errmsg)
            LOGGER.error(errmsg)
        # Don't stop beam capture. Only stop at the end of the test.
        #try:
        #    LOGGER.info("Issue {} capture stop via CAM int".format(beam))
        #    reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam)
        #    errmsg = "Failed to issue capture_stop for beam {}: {}".format(beam, str(reply))
        #    assert reply.reply_ok(), errmsg
        #except AssertionError:
        #    Aqf.failed(errmsg)
        #    LOGGER.exception(errmsg)

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
                    # if element.name.find('captured_timestamps') > -1:
                    #     cap_ts = np.array(element.value)
                    if element.name.find("bf_raw") > -1:
                        bf_raw = np.array(element.value)
                    elif element.name.find("timestamps") > -1:
                        bf_ts = np.array(element.value)
                    elif element.name.find("flags") > -1:
                        bf_flags = np.array(element.value)
            os.remove(newest_f)
            return bf_raw, bf_flags, bf_ts, in_wgts


    @staticmethod
    def populate_beam_dict(num_wgts_to_set, value, beam_dict):
        """
            If num_wgts_to_set = -1 all inputs will be set
        """
        beam_dict = dict.fromkeys(beam_dict, 0)
        ctr = 0
        for key in beam_dict:
            if ctr < num_wgts_to_set or num_wgts_to_set == -1:
                beam_dict[key] = value
                ctr += 1
        return beam_dict


    @staticmethod
    def populate_beam_dict_idx(index, value, beam_dict):
        """
            Set specified beam index to weight, all other values to 0
        """
        for key in beam_dict:
            key_idx = int(filter(str.isdigit, key))
            if key_idx == index:
                beam_dict[key] = value
            else:
                beam_dict[key] = 0
        return beam_dict


    def set_beam_quant_gain(self, beam, gain):
        try:
            reply, informs = self.corr_fix.katcp_rct.req.beam_quant_gains(beam, gain, timeout=60)
            assert reply.reply_ok()
            actual_beam_gain = float(reply.arguments[1])
            msg = (
                "Requested beamformer level adjust gain of {:.2f}, "
                "actual gain set to {:.2f}.".format(gain, actual_beam_gain)
            )
            Aqf.almost_equals(actual_beam_gain, gain, 0.1, msg)
            return actual_beam_gain
        except Exception:
            Aqf.failed("Failed to set beamformer quantiser gain via CAM interface")


    def get_hosts(self, hosts=None, sensor="hostname-functional-mapping"):
        """
        Get list of f/xhosts from sensors or config
        return:
        list
        """
        for i in range(5):
            try:
                assert hosts
                reply, informs = self.katcp_req_sensors.sensor_value(sensor)
                assert reply.reply_ok()
                informs = eval(informs[0].arguments[-1])
                informs = dict((val, key) for key, val in informs.iteritems())
                return [v for i, v in informs.iteritems() if i.startswith(hosts)]
            except SyntaxError:
                self.logger.warn('{} not populated, Waiting 20s and retrying, '
                        'on retry {}'.format(sensor,i))
                time.sleep(20)
            except AssertionError:
                if hosts.startswith("fhost"):
                    engine = self.corr_fix.corr_config.get("fengine")
                else:
                    engine = self.corr_fix.corr_config.get("xengine")
                return engine.get("hosts", [])


    def get_gain_all(self):
        """
        Retrieve gain of all inputs via sensors
        """
        try:
            for i in range(4):
                try:
                    reply, informs = self.katcp_req.sensor_value()
                    self.assertTrue(reply.reply_ok())
                    break
                except AssertionError:
                    self.logger.warn('Sensors not received, Waiting 20s and retrying')
                    time.sleep(20)
            assert reply.reply_ok()
        except AssertionError:
            self.Failed("Failed to retrieve sensors after 4 retries")
            return
        else:
            sensors_required = []
            labels = [x.lower() for x in self.cam_sensors.input_labels]
            if self.conf_file['instrument_params']['sensor_named_by_label'] == 'False':
                num_inputs = len(labels)
                labels = ['input'+str(x) for x in range(num_inputs)]
            for label in labels:
                sens_name = self.corr_fix.feng_product_name+'-'+label+'-eq'
                sensors_required.append(sens_name)
            search = re.compile('|'.join(sensors_required))
            sensors = OrderedDict()
            for inf in informs:
                if search.match(inf.arguments[2]):
                    try:
                        sensors[inf.arguments[2].replace('-', '_')] = eval(inf.arguments[4])
                    except Exception:
                        sensors[inf.arguments[2].replace('-', '_')] = inf.arguments[4]
            return list(set(flatten(sensors.values())))[0]


    def get_fftshift_all(self):
        """
        Retrieve gain of all inputs via sensors
        """
        try:
            for i in range(4):
                try:
                    reply, informs = self.katcp_req.sensor_value(timeout=cam_timeout)
                    self.assertTrue(reply.reply_ok())
                    break
                except AssertionError:
                    self.logger.warn('Sensors not received, Waiting 20s and retrying: '
                            '{}'.format(reply))
                    time.sleep(20)
            assert reply.reply_ok()
        except AssertionError:
            self.Failed("Failed to retrieve sensors after 4 retries")
            return
        else:
            sensors_required = []
            labels = [x.lower() for x in self.cam_sensors.input_labels]
            if self.conf_file['instrument_params']['sensor_named_by_label'] == 'False':
                num_inputs = len(labels)
                labels = ['input'+str(x) for x in range(num_inputs)]
            for label in labels:
                sens_name = self.corr_fix.feng_product_name+'-'+label+'-fft0-shift'
                sensors_required.append(sens_name)
            search = re.compile('|'.join(sensors_required))
            sensors = OrderedDict()
            for inf in informs:
                if search.match(inf.arguments[2]):
                    try:
                        sensors[inf.arguments[2].replace('-', '_')] = eval(inf.arguments[4])
                    except Exception:
                        sensors[inf.arguments[2].replace('-', '_')] = inf.arguments[4]
            return list(set(flatten(sensors.values())))[0]


    def create_logs_directory(self):
        """
        Create custom `logs_instrument` directory on the test dir to store generated images
        param: self
        """
        test_dir, test_name = os.path.split(os.path.dirname(os.path.realpath(__file__)))
        path = test_dir + "/logs_" + self.corr_fix.instrument
        if not os.path.exists(path):
            LOGGER.info("Created %s for storing images." % path)
            os.makedirs(path)
        return path


    def confirm_out_dest_ip(self):
        """Confirm is correlators output destination ip is the same as the one in config file
        :param: Object
        :rtype: Boolean
        """
        parse_address = StreamAddress._parse_address_string
        try:
            xhosts = self.get_hosts("xhosts")
            xhost = xhosts[random.randrange(xhosts)]
            int_ip = int(xhost.registers.gbe_iptx.read()["data"]["reg"])
            xhost_ip = inet_ntoa(pack(">L", int_ip))
            dest_ip = list(
                parse_address(self.correlator.configd["xengine"]["output_destinations_base"])
            )[0]
            assert dest_ip == xhost_ip
            return True
        except Exception:
            LOGGER.exception("Failed to retrieve correlator ip address.")
            return False


    def restore_src_names(self):
        """Restore default CBF input/source names.
        :param: Object
        :rtype: Boolean
        """
        try:
            orig_src_names = self.correlator.configd["fengine"]["source_names"].split(",")
        except Exception:
            orig_src_names = ["ant_{}".format(x) for x in xrange(self.correlator.n_antennas * 2)]

        LOGGER.info("Restoring source names to %s" % (", ".join(orig_src_names)))
        try:
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*orig_src_names)
            assert reply.reply_ok()
            self.corr_fix.issue_metadata
            LOGGER.info(
                "Successfully restored source names back to default %s" % (", ".join(orig_src_names))
            )
        except Exception:
            LOGGER.exception("Failed to restore CBF source names back to default.")
            return False


    def set_default_eq(self):
        """ Iterate through config sources and set eq's as per config file
        Param: Correlator: Object
        Return: None
        """
        try:
            eq_levels = complex(self.corr_fix.configd.get("fengine").get("default_eq_poly"))
            reply, informs = self.corr_fix.katcp_rct.req.gain_all(eq_levels, timeout=cam_timeout)
            assert reply.reply_ok()
            LOGGER.info("Reset gains to default values from config file.\n")
            return True
        except Exception:
            LOGGER.exception("Failed to set gains on all inputs with %s " % (eq_levels))
            return False


    #TODO This does nothing... check this code
    def restore_initial_equalisations(self):
        return
        init_eq = self.get_gain_all()
        try:
            reply, informs = self.corr_fix.katcp_rct.req.gain_all(init_eq, timeout=cam_timeout)
            assert reply.reply_ok()
            return True
        except Exception:
            msg = "Failed to set gain for all inputs with gain of %s" % init_eq
            LOGGER.exception(msg)
            return False
        self.addCleanup(self.restore_initial_equalisations)


    def get_baselines_lookup(self, test_input=None, auto_corr_index=False, sorted_lookup=False):
        """Get list of all the baselines present in the correlator output.
        Param:
            self: object
            Return: dict:
            baseline lookup with tuple of input label strings keys
            `(bl_label_A, bl_label_B)` and values bl_AB_ind, the index into the
            correlator dump's baselines
        """
        try:
            bls_ordering = eval(self.cam_sensors.get_value("bls_ordering"))
            LOGGER.info("Retrieved bls ordering via CAM Interface")
        except Exception:
            bls_ordering = None
            LOGGER.exception("Failed to retrieve bls ordering from CAM int.")
            return

        baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(bls_ordering)}
        if auto_corr_index:
            for idx, val in enumerate(baseline_lookup):
                if val[0] == test_input and val[1] == test_input:
                    auto_corr_idx = idx
            return [baseline_lookup, auto_corr_idx]
        elif sorted_lookup:
            return sorted(baseline_lookup.items(), key=operator.itemgetter(1))
        else:
            return baseline_lookup
    
    def get_test_levels(self, profile='noise'):
        """
        Get the default instrument gains/fft shift and digitiser levels
        for the requested intrument as defined in the test config file.
        param:
            self: Object
                correlator_fixture object
            profile: String
                Continious Wave or Noise profile ("cw","noise") 
        returns:
            awgn_scale: Float
                Dsim noise scale
            cw_scale: Float
                Dsim cw scale
            gain: String
                Complex equaliser gain
            fft_shift: Int
                FFT shift
        """

        int_time = round(self.cam_sensors.get_value("int_time"),1)
        int_time = str(int_time).split('.')
        int_time = '_'+ int_time[0] + '_' + int_time[1]
        if (profile in ('noise','cw')):
            try:
                if "54M32k" in self.instrument:
                    awgn_scale = self.corr_fix._test_config_file["instrument_params"]["{}32knbh_awgn_scale".format(profile)]
                    cw_scale   = self.corr_fix._test_config_file["instrument_params"]["{}32knbh_cw_scale".format(profile)]
                    gain       = self.corr_fix._test_config_file["instrument_params"]["{}32knbh_gain{}".format(profile, int_time)]
                    fft_shift  = self.corr_fix._test_config_file["instrument_params"]["{}32knbh_fft_shift".format(profile)]
                elif "107M32k" in self.instrument:
                    awgn_scale = self.corr_fix._test_config_file["instrument_params"]["{}32knbf_awgn_scale".format(profile)]
                    cw_scale   = self.corr_fix._test_config_file["instrument_params"]["{}32knbf_cw_scale".format(profile)]
                    gain       = self.corr_fix._test_config_file["instrument_params"]["{}32knbf_gain{}".format(profile, int_time)]
                    fft_shift  = self.corr_fix._test_config_file["instrument_params"]["{}32knbf_fft_shift".format(profile)]
                elif "1k" in self.instrument:
                    awgn_scale = self.corr_fix._test_config_file["instrument_params"]["{}1k_awgn_scale".format(profile)]
                    cw_scale   = self.corr_fix._test_config_file["instrument_params"]["{}1k_cw_scale".format(profile)]
                    gain       = self.corr_fix._test_config_file["instrument_params"]["{}1k_gain{}".format(profile, int_time)]
                    fft_shift  = self.corr_fix._test_config_file["instrument_params"]["{}1k_fft_shift".format(profile)]
                elif "4k" in self.instrument:                                     
                    awgn_scale = self.corr_fix._test_config_file["instrument_params"]["{}4k_awgn_scale".format(profile)]
                    cw_scale   = self.corr_fix._test_config_file["instrument_params"]["{}4k_cw_scale".format(profile)]
                    gain       = self.corr_fix._test_config_file["instrument_params"]["{}4k_gain{}".format(profile, int_time)]
                    fft_shift  = self.corr_fix._test_config_file["instrument_params"]["{}4k_fft_shift".format(profile)]
                elif "32k" in self.instrument:
                    awgn_scale = self.corr_fix._test_config_file["instrument_params"]["{}32k_awgn_scale".format(profile)]
                    cw_scale   = self.corr_fix._test_config_file["instrument_params"]["{}32k_cw_scale".format(profile)]
                    gain       = self.corr_fix._test_config_file["instrument_params"]["{}32k_gain{}".format(profile, int_time)]
                    fft_shift  = self.corr_fix._test_config_file["instrument_params"]["{}32k_fft_shift".format(profile)]
                else:
                    msg = "Instrument not found: {}".format(self.instrument)
                    self.logger.exception(msg)
                    self.Failed(msg)
                    return False
            except KeyError:
                msg = ('Profile values for integration time {} does not exist. Fix site_test_conf file.'.format(int_time))
                self.Failed(msg)
                return False
        else:
            msg = "Profile selected does not exist: {}".format(profile)
            self.logger.exception(msg)
            self.Failed(msg)
            return False
        # Scale gain according to profile_acc_time
        #gain_real = float(gain.split('+')[0])
        #gain_imag = gain.split('+')[1]
        ##gain_real = gain_real / (int_time / float(self.conf_file["instrument_params"]["profile_acc_time"]))
        #gain_real = gain_real / 2
        #gain_real = round(gain_real,1)
        #gain = '{}+{}'.format(gain_real,gain_imag)
        return float(awgn_scale), float(cw_scale), gain, fft_shift

    def set_input_levels(self, awgn_scale=None, cw_scale=None, freq=None, output_scale=None, fft_shift=None,
                        gain=None, cw_src=0, corr_noise=True):
        """
        Set the digitiser simulator (dsim) output levels, FFT shift
        and quantiser gain to optimum levels - Hardcoded.
        Param:
            self: Object
                correlator_fixture object
            awgn_scale : Float
                Gaussian noise digitiser output scale.
            cw_scale: Float
                constant wave digitiser output scale.
            freq: Float
                arbitrary frequency to set with the digitiser simulator
            fft_shift: Int
                current FFT shift value
            gain: Complex/Str
                quantiser gain value
            cw_src: Int
                source 0 or 1
        Return: Bool
        """
        # Select dsim mux
        self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
        self.dhost.registers.src_sel_cntrl.write(src_sel_1=0)
        # Zero everything
        self.dhost.noise_sources.noise_corr.set(scale=0)
        self.dhost.noise_sources.noise_0.set(scale=0)
        self.dhost.noise_sources.noise_1.set(scale=0)
        self.dhost.sine_sources.sin_corr.set(frequency=0, scale=0)
        self.dhost.sine_sources.sin_0.set(frequency=0, scale=0)
        self.dhost.sine_sources.sin_1.set(frequency=0, scale=0)
        self.dhost.outputs.out_0.scale_output(1.0)
        self.dhost.outputs.out_1.scale_output(1.0)
        time.sleep(0.1)
        if cw_scale is not None:
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=cw_scale)

        if awgn_scale is not None:
            if corr_noise:
                self.dhost.noise_sources.noise_corr.set(scale=awgn_scale)
            else:
                self.dhost.noise_sources.noise_0.set(scale=awgn_scale)
                self.dhost.noise_sources.noise_1.set(scale=awgn_scale)

        if output_scale is not None:
            self.dhost.outputs.out_0.scale_output(output_scale)
            self.dhost.outputs.out_1.scale_output(output_scale)

        def set_fft_shift(self):
            try:
                start_time = time.time()
                reply, _informs = self.corr_fix.katcp_rct.req.fft_shift(fft_shift, timeout=cam_timeout)
                cmd_time = time.time()-start_time
                assert reply.reply_ok()
                LOGGER.info("F-Engines FFT shift set to {} via CAM interface, "
                        "cmd took {}s".format(fft_shift, cmd_time))

                return True
            except Exception:
                LOGGER.exception("Failed to set FFT shift via CAM interface")
                return False

        LOGGER.info("Setting desired FFT-Shift via CAM interface.")
        if set_fft_shift(self) is not True:
            LOGGER.error("Failed to set FFT-Shift via CAM interface")

        sources = self.cam_sensors.input_labels
        source_gain_dict = dict(ChainMap(*[{i: "{}".format(gain)} for i in sources]))
        try:
            LOGGER.info("Setting desired gain/eq via CAM interface.")
            eq_level = list(set(source_gain_dict.values()))
            if len(eq_level) != 1:
                for i, v in source_gain_dict.items():
                    LOGGER.info("Input %s gain set to %s" % (i, v))
                    start_time = time.time()
                    reply, informs = self.corr_fix.katcp_rct.req.gain(i, v, timeout=cam_timeout)
                    cmd_time = time.time()-start_time
                    LOGGER.info("Setting gain levels via loop took {}s".format(cmd_time))
                    assert reply.reply_ok()
            else:
                eq_level = eq_level[0]
                start_time = time.time()
                LOGGER.info("Setting gain levels to all inputs to %s" % (eq_level))
                reply, informs = self.corr_fix.katcp_rct.req.gain_all(eq_level, timeout=cam_timeout)
                cmd_time = time.time()-start_time
                LOGGER.info("Setting gain levels via gain-all took {}s".format(cmd_time))
                assert reply.reply_ok()
            LOGGER.info("Gains set successfully")
            return True
        except Exception:
            LOGGER.exception("Failed to set gain for input.")
            return False


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
        ve_desc = _results.get("Verification Event Description", "TBD")
        Aqf.procedure(r"%s" % ve_desc)
        try:
            assert evaluate(os.getenv("MANUAL_TEST", "False")) or evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            results = r"%s" % _results.get("Verification Event Results", "TBD")
            if results != "TBD":
                self.Step(r"%s" % _results.get("Verification Requirement Description", "TBD"))
                self.Passed(r"%s" % results)
                perf = _results.get("Verification Event Performed By", "TBD")
                _date = _results.get("Date of Verification Event", "TBD")
                if perf != "TBD":
                    Aqf.hop(r"Test run by: %s on %s" % (perf, _date))
            else:
                Aqf.tbd("This test results outstanding.")


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
        self, profile="noise", cw_freq=0, cw_src=0, cw_margin=0.05, trgt_bits=3.5, trgt_q_std=0.30,
        fft_shift=511):
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
                reply, _ = self.katcp_req.gain(source, gain_str)
                self.assertTrue(reply.reply_ok())
                assert reply.arguments[1:][0] == gain_str
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
            ch_bw = self.cam_sensors.ch_center_freqs[1]
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
            reply, _ = self.katcp_req.fft_shift(fft_shift)
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
                baseline_lookup = self.get_baselines_lookup(dump)
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


    def _systems_tests(self):
        """Checking system stability before and after use"""
        try:
            self.Step("Checking system sensors integrity.")
            try:
                for i in range(4):
                    try:
                        reply, informs = self.katcp_req.sensor_value()
                        self.assertTrue(reply.reply_ok())
                        break
                    except AssertionError:
                        self.logger.warn('Sensors not received, Waiting 20s and retrying')
                        time.sleep(20)
                assert reply.reply_ok()
            except AssertionError:
                self.Failed("Failed to retrieve sensors after 4 retries")
                return
            #for i in range(1):
            #    try:
            #        reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=30)
            #    except Exception:
            #        reply, informs = self.katcp_req.sensor_value(timeout=30)
            #    time.sleep(10)

            _errored_sensors_ = ", ".join(
                sorted(list(set([i.arguments[2] for i in informs if "error" in i.arguments[-2]])))
            )
            _warning_sensors_ = ", ".join(
                sorted(list(set([i.arguments[2] for i in informs if "warn" in i.arguments[-2]])))
            )
        except Exception:
            self.Note("Could not retrieve sensors via CAM interface.")
        else:
            if _errored_sensors_:
                self.Note("The following number of sensors (%s) have `ERRORS`: %s" % (
                    len(_errored_sensors_.split(',')), _errored_sensors_))
                # print('Following sensors have ERRORS: %s' % _errored_sensors_)
            #if _warning_sensors_:
            #    self.Note("The following number of sensors (%s) have `WARNINGS`: %s" % (
            #        len(_warning_sensors_.split(',')), _warning_sensors_))
            #    # print('Following sensors have WARNINGS: %s' % _warning_sensors_)


    def _delays_setup(self, test_source_idx=(0,1), determine_start_time=True,
                      awgn_scale_override=None,
                      gain_override=None,
                      gain_multiplier = None):
        # Put some correlated noise on both outputs
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
        if awgn_scale_override: awgn_scale = awgn_scale_override
        if gain_override: gain = gain_override
        if gain_multiplier: gain = complex(gain)*gain_multiplier

        self.Step("Configure digitiser simulator to generate Gaussian noise.")
        self.Progress(
            "Digitiser simulator configured to generate Gaussian noise with scale: {}, "
            "gain: {} and fft shift: {}.".format(awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale,
            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        # local_src_names = self.cam_sensors.custom_input_labels
        network_latency = float(self.conf_file["instrument_params"]["network_latency"])
        cam_max_load_time = int(self.conf_file["instrument_params"]["cam_max_load_time"])
        source_names = self.cam_sensors.input_labels
        # Get name for test_source_idx
        test_source = source_names[test_source_idx[0]]
        ref_source = source_names[test_source_idx[1]]
        num_inputs = len(source_names)
        # Number of integrations to load delays in the future
        int_time = self.cam_sensors.get_value("int_time")
        delay_load_lead_time = float(self.conf_file['instrument_params']['delay_load_lead_time'])
        min_int_delay_load = math.ceil(delay_load_lead_time / int_time)
        num_int_delay_load = int(self.conf_file['instrument_params']['num_int_delay_load'])
        num_int_delay_load = max(num_int_delay_load, min_int_delay_load)
        load_lead_time = num_int_delay_load * int_time
        self.Step("Clear all coarse and fine delays for all inputs before test commences.")
        delays_cleared = self.clear_all_delays()
        if not delays_cleared:
            self.Failed("Delays were not completely cleared, data might be corrupted.")
        else:
            self.Passed("Cleared all previously applied delays prior to test.")

        self.logger.info("Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.")
        try:
            #initial_dump = self.receiver.get_clean_dump()
            initial_dump = self.get_real_clean_dump()
        except Queue.Empty:
            errmsg = "Could not retrieve clean SPEAD accumulation: Queue might be Empty."
            self.Failed(errmsg, exc_info=True)
        else:
            self.logger.info("Successfully retrieved initial spead accumulation")
            sync_epoch = self.cam_sensors.get_value("sync_epoch")
            # n_accs = self.cam_sensors.get_value('n_accs')]
            # no_chans = range(self.n_chans_selected)
            time_stamp = initial_dump["timestamp"]
            dump_ts    = initial_dump["dump_timestamp"]
            # ticks_between_spectra = initial_dump['ticks_between_spectra'].value
            # int_time_ticks = n_accs * ticks_between_spectra
            curr_time = time.time()
            curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
            # If dump timestamp is behind add to load leadtime
            # TODO: this might have caused delay tracking inaccuracy, take server time out of the calc
            #load_delta = np.floor(curr_time-dump_ts)/int_time
            load_delta = (int((curr_time-dump_ts)/int_time)+1)*int_time
            if load_delta < 0:
                self.Failed("Current CMC time {} is behind dump timestamp {}, re-synchronise instrument.")
                return False
            t_apply = initial_dump["dump_timestamp"] + load_lead_time + load_delta
            t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
            baseline_lookup = self.get_baselines_lookup()
            # Choose baseline for phase comparison
            swap = False
            while True:
                try:
                    if not swap:
                        baseline_index = baseline_lookup[(ref_source, test_source)]
                        break
                    else:
                        baseline_index = baseline_lookup[(test_source, ref_source)]
                        break
                except KeyError:
                    if not swap:
                        swap = True
                    else:
                        self.Failed("Initial SPEAD accumulation does not contain correct baseline ordering format.")
                        return False
            #self.Step("Get list of all the baselines present in the correlator output")
            self.Progress(
                "Selected baseline {} ({},{}) for testing.".format(baseline_index, test_source, ref_source)
            )
            if determine_start_time:
                self.Progress(
                    "Time to apply delays: %s (%s), Current cmc time: %s (%s), Delays will be "
                    "applied %s integrations/accumulations in the future."
                    % (t_apply, t_apply_readable, curr_time, curr_time_readable, num_int_delay_load)
                )
            return {
                "baseline_index": baseline_index,
                "baseline_lookup": baseline_lookup,
                "initial_dump": initial_dump,
                "int_time": int_time,
                "network_latency": network_latency,
                "num_inputs": num_inputs,
                "sample_period": self.cam_sensors.sample_period,
                "t_apply": t_apply,
                "test_source": test_source,
                "test_source_ind": test_source_idx[0],
                "time_stamp": time_stamp,
                "sync_epoch": sync_epoch,
                "num_int": num_int_delay_load,
                "cam_max_load_time": cam_max_load_time,
            }

    def _confirm_delays(self, delay_coefficients, err_margin = 1):
        delay_coeff = [re.split(',|:', x) for x in delay_coefficients]
        delay_coeff = [[float(y) for y in x] for x in delay_coeff]
        #return True, delay_coeff
        labels = [x.lower() for x in self.cam_sensors.input_labels]
        if self.conf_file['instrument_params']['sensor_named_by_label'] == 'False':
            num_inputs = len(labels)
            labels = ['input'+str(x) for x in range(num_inputs)]
        retry = 5
        errmsg = ""
        while retry:
            delay_values = []
            for label in labels:
                sens_name = self.corr_fix.feng_product_name+'-'+label+'-delay'
                sens_name = sens_name.replace('-','_')
                delay_value = (self.cam_sensors.get_value(sens_name))
                delay_values.append(delay_value)
            try:
                delay_values = [x[1:-1] for x in delay_values]
                delay_values = [x.split(',') for x in delay_values]
                delay_values = [x[1:] for x in delay_values]
                delay_values = [[float(y) for y in x] for x in delay_values]
            except:
                errmsg = "Read delay values are not in the correct format."
                retry -= 1
            else:
                diff_array = np.abs(np.array(delay_values) - np.array(delay_coeff))
                coeff_err = (diff_array > err_margin).any()
                if coeff_err:
                    errmsg = ('Actual delay values set not within tolerance:\n'
                              'Actual: {}\n'
                              'Requested: {}'
                              ''.format(delay_values, delay_coeff))
                    retry -= 1
                else:
                    return not(coeff_err), delay_values
            self.logger.warn('Waiting 60 and retrying because of msg: {}'.format(errmsg))
            time.sleep(60)
        self.Error(errmsg, exc_info=True)
        return False, None

        #error_margin = np.deg2rad(err_margin)
        #delay_name_pos = ['delay', 'delay rate', 'phase offset', 'phase rate']
        #for idx, row in enumerate(diff_values):
        #    err_col = error_indexes[1][err_idx]
        #    actual_delay_val  = actual_delay_coef[err_row][err_col]
        #    request_delay_val = request_delay_coef[err_row][err_col]
        #    diff = actual_delay_val - request_delay_val
        #    if abs(diff) > error_margin:
        #        self.Step('{}'.format(error_margin))
        #        setting = delay_name_pos[err_col]
        #        errmsg = ('Input {} {} set ({:.5f}) does not match requested ({:.5f}), '
        #                  'difference = {}'
        #                  ''.format(err_row, setting, actual_delay_val,
        #                            request_delay_val, diff))
        #        self.Failed(errmsg, exc_info = True)


    def _test_coeff(self, setup_data, delay_coefficients, max_wait_dumps=50):
        reply, _informs = self.katcp_req.delays("antenna-channelised-voltage",
            setup_data["t_apply"], *delay_coefficients, timeout=30)
        n_ants = int(self.cam_sensors.get_value("n_ants"))
        actual_delay_coef = []
        for inputs in range(n_ants*2):
            delays = self.cam_sensors.get_value('input{}_delay'.format(inputs))
            delays = delays.replace('(','')
            delays = delays.replace(')','')
            delays = delays.split(',')
            try:
                delays = map(float, delays)
                actual_delay_coef.append(delays)
            except ValueError:
                errmsg = "Error reading delay values: {}".format(delays)
                self.Failed(errmsg, exc_info=True)
        actual_delay_coef  = np.array(actual_delay_coef)
        actual_delay_coef  = np.delete(actual_delay_coef, 0, axis=1)
        request_delay_coef = [s.replace(':',',') for s in delay_coefficients]
        request_delay_coef = [s.split(',') for s in request_delay_coef]
        request_delay_coef = [map(float,inp) for inp in request_delay_coef]
        error_indexes = np.where(request_delay_coef != actual_delay_coef)
        error_margin = np.deg2rad(1)
        delay_name_pos = ['delay', 'delay rate', 'phase offset', 'phase rate']
        for err_idx, err_row in enumerate(error_indexes[0]):
            err_col = error_indexes[1][err_idx]
            actual_delay_val  = actual_delay_coef[err_row][err_col]
            request_delay_val = request_delay_coef[err_row][err_col]
            diff = actual_delay_val - request_delay_val
            if abs(diff) > error_margin:
                setting = delay_name_pos[err_col]
                errmsg = ('Input {} {} set ({:.5f}) does not match requested ({:.5f}), '
                          'difference = {}'
                          ''.format(err_row, setting, actual_delay_val, 
                                    request_delay_val, diff))
                self.Failed(errmsg, exc_info = True)


    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients, 
                         max_wait_dumps=30, save_filename = None):
        try:
            self.logger.info("Request Fringe/Delay(s) Corrections via CAM interface.")
            load_strt_time = time.time()
            reply, _informs = self.katcp_req.delays(self.corr_fix.feng_product_name,
                setup_data["t_apply"], *delay_coefficients, timeout=30)
            load_done_time = time.time()
            errmsg = ("%s: Failed to set delays via CAM interface with load-time: %s, "
                      "Delay coefficients: %s" % (
                            str(reply).replace("\_", " "),
                            setup_data["t_apply"],
                            delay_coefficients,
                        ))
            self.assertTrue(reply.reply_ok(), errmsg)
            if "updated" not in reply.arguments[1]:
                errmsg = errmsg + ' katcp reply: {}'.format(reply)
                raise AssertionError()
            cmd_load_time = round(load_done_time - load_strt_time, 3)
            self.Step("Fringe/Delay load command took {} seconds".format(cmd_load_time))
            # _give_up = int(setup_data['num_int'] * setup_data['int_time'] * 3)
            # while True:
            #    _give_up -= 1
            #    try:
            #        self.logger.info('Waiting for the delays to be updated on sensors: %s retry' % _give_up)
            #        try:
            #            reply_, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
            #        except:
            #            reply_, informs = self.katcp_req.sensor_value()
            #        self.assertTrue(reply_.reply_ok())
            #    except Exception:
            #        self.logger.exception('Weirdly I could not get the sensor values')
            #    else:
            #        delays_updated = list(set([int(i.arguments[-1]) for i in informs
            #                                if '.cd.delay' in i.arguments[2]]))[0]
            #        if delays_updated:
            #            self.logger.info('Delays have been successfully set')
            #            msg = ('Delays set successfully via CAM interface: reply %s' % str(reply))
            #            self.Passed(msg)
            #            break
            #    if _give_up == 0:
            #        msg = ("Could not confirm the delays in the time stipulated, exiting")
            #        self.logger.error(msg)
            #        self.Failed(msg)
            #        break
            #    time.sleep(1)

            # Tested elsewhere
            # cam_max_load_time = setup_data['cam_max_load_time']
            # msg = 'Time it took to load delay/fringe(s) %s is less than %ss' % (cmd_load_time,
            #        cam_max_load_time)
            # Aqf.less(cmd_load_time, cam_max_load_time, msg)
        except AssertionError:
            self.Failed(errmsg, exc_info=True)
            return

        last_discard = setup_data["t_apply"] - setup_data["int_time"]
        num_discards = 0
        fringe_dumps = []
        self.Step(
            "Getting SPEAD accumulation containing the change in fringes(s) on input: %s "
            "baseline: %s, and discard all preceding accumulations."
            % (setup_data["test_source"], setup_data["baseline_index"])
        )
        while True:
            num_discards += 1
            try:
                dump = self.receiver.data_queue.get()
                self.assertIsInstance(dump, dict)
            except Exception:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue might be Empty."
                self.Failed(errmsg, exc_info=True)
            else:
                time_diff = np.abs(dump["dump_timestamp"] - last_discard)
                if time_diff < 0.1 * setup_data["int_time"]:
                    # Dont add this dump... delay has not been applied yet...
                    #fringe_dumps.append(dump)
                    msg = (
                        "Received final accumulation before fringe "
                        "application with dump timestamp: %s, apply time: %s "
                        "(Difference %s)" % (dump["dump_timestamp"], setup_data["t_apply"], time_diff)
                    )
                    self.Passed(msg)
                    self.logger.info(msg)
                    break

                if num_discards > max_wait_dumps:
                    self.Failed(
                        "Could not get accumulation with correct timestamp within %s "
                        "accumulation periods." % max_wait_dumps
                    )
                    # break
                    return
                else:
                    msg = (
                        "Discarding (#%d) Spead accumulation with dump timestamp: %s"
                        ", apply time: %s "
                        "(Difference %.2f), Current cmc time: %s."
                        % (
                            num_discards, dump["dump_timestamp"],
                            setup_data["t_apply"], time_diff, time.time())
                    )
                    self.Progress(msg)
                    #if num_discards <= 2:
                    #    self.Progress(msg)
                    #elif num_discards == 3:
                    #    self.Progress("...")
                    #elif time_diff < 3:
                    #    self.Progress(msg)

        for i in range(dump_counts):
            self.Progress("Getting subsequent SPEAD accumulation {}.".format(i + 1))
            try:
                dump = self.receiver.data_queue.get()
                self.assertIsInstance(dump, dict)
            except Exception:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue might be Empty."
                self.Error(errmsg, exc_info=True)
            else:
                fringe_dumps.append(dump)

        if save_filename:
            with open(save_filename, 'w') as f:
                np.save(f, fringe_dumps)

        chan_resp = []
        phases = []
        for acc in fringe_dumps:
            dval = acc["xeng_raw"]
            freq_response = normalised_magnitude(dval[:, setup_data["baseline_index"], :])
            chan_resp.append(freq_response)
            data = complexise(dval[:, setup_data["baseline_index"], :])
            phases.append(np.angle(data))
        return zip(phases, chan_resp), fringe_dumps

    def _get_expected_data(self, setup_data, dump_counts, delay_coefficients, actual_phases, save_filename=None):
        def calc_actual_delay(setup_data):
            no_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            first_dump = np.unwrap(actual_phases[0])
            actual_slope = np.polyfit(range(0, no_ch), first_dump, 1)[0] * no_ch
            actual_delay = self.cam_sensors.sample_period * actual_slope / np.pi
            return actual_delay

        def gen_delay_vector(delay, setup_data):
            res = []
            no_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            delay_slope = np.pi * (delay / self.cam_sensors.sample_period)
            c = delay_slope / 2
            for i in range(0, no_ch):
                m = i / float(no_ch)
                res.append(delay_slope * m - c)
            return res

        def gen_delay_data(delay, delay_rate, dump_counts, setup_data):
            expected_phases = []
            prev_delay_rate = 0
            for dump in range(0, dump_counts):
                # For delay rate the expected delay is the average of delays
                # applied during the integration. This is equal to the
                # delay delta over the integration divided by two
                max_delay_rate = dump * delay_rate
                avg_delay_rate = ((max_delay_rate - prev_delay_rate) / 2) + prev_delay_rate
                prev_delay_rate = max_delay_rate
                tot_delay = delay + avg_delay_rate * self.cam_sensors.get_value("int_time")
                expected_phases.append(gen_delay_vector(tot_delay, setup_data))
            return expected_phases

        def calc_actual_offset(setup_data):
            # mid_ch = no_ch / 2
            first_dump = actual_phases[0]
            # Determine average offset around 5 middle channels
            actual_offset = np.average(first_dump)  # [mid_ch-3:mid_ch+3])
            return actual_offset

        def gen_fringe_vector(offset, setup_data):
            return [offset] * self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")

        def gen_fringe_data(fringe_offset, fringe_rate, dump_counts, setup_data):
            expected_phases = []
            prev_fringe_rate = 0
            for dump in range(0, dump_counts):
                # For fringe rate the expected delay is the average of delays
                # applied during the integration. This is equal to the
                # delay delta over the integration divided by two
                max_fringe_rate = dump * fringe_rate
                avg_fringe_rate = ((max_fringe_rate - prev_fringe_rate) / 2) + prev_fringe_rate
                prev_fringe_rate = max_fringe_rate
                offset = -(fringe_offset + avg_fringe_rate * self.cam_sensors.get_value("int_time"))
                expected_phases.append(gen_fringe_vector(offset, setup_data))
            return expected_phases

        try:
            if type(delay_coefficients[0]) == str:
                ant_delay = [s.replace(':',',') for s in delay_coefficients]
                ant_delay = [s.split(',') for s in ant_delay]
                ant_delay = [map(float,inp) for inp in ant_delay]
            else:
                ant_delay = delay_coefficients

            ant_idx = setup_data["test_source_ind"]
            delay = ant_delay[ant_idx][0]
            delay_rate = ant_delay[ant_idx][1]
            fringe_offset = ant_delay[ant_idx][2]
            fringe_rate = ant_delay[ant_idx][3]
        except Exception as e:
            raise ValueError("[} is not a valid delay setting: {}".format(ant_delay, e))

        delay_data = np.array((gen_delay_data(delay, delay_rate, dump_counts + 1, setup_data)))[1:]
        fringe_data = np.array(gen_fringe_data(fringe_offset, fringe_rate, dump_counts + 1, setup_data))[1:]
        decimation_factor = float(self.cam_sensors.get_value("decimation_factor"))
        delay_data = delay_data/decimation_factor
        result = (delay_data + fringe_data)
        wrapped_results = (result + np.pi) % (2 * np.pi) - np.pi
        # Cut the selected channel slice
        wrapped_results = wrapped_results[:,self.start_channel:self.stop_channel]
        if save_filename:
            save_filename = save_filename[:-4] + "_exp.npy"
            Aqf.hop('Saving raw delay data: {}'.format(save_filename.split('/')[-1]))
            with open(save_filename, 'w') as f:
                np.save(f, wrapped_results)

        if (fringe_offset or fringe_rate) != 0:
            fringe_phase = [np.abs((np.min(phase) + np.max(phase)) / 2.0) for phase in fringe_data]
            return zip(fringe_phase, wrapped_results)
        else:
            delay_phase = [np.abs((np.min(phase) - np.max(phase)) / 2.0) for phase in delay_data]
            return zip(delay_phase, wrapped_results)


    def _process_power_log(self, start_timestamp, power_log_file):
        max_power_per_rack = 6.25
        max_power_diff_per_rack = 33
        max_power_cbf = 60
        time_gap = 60

        df = pd.read_csv(power_log_file, delimiter="\t")
        headers = list(df.keys())
        exp_headers = ["Sample Time", "PDU Host", "Phase Current", "Phase Power"]
        if headers != exp_headers:
            raise IOError(power_log_file)
        pdus = list(set(list(df[headers[1]])))
        # Slice out requested time block
        end_ts = df["Sample Time"].iloc[-1]
        try:
            strt_idx = df[df["Sample Time"] >= int(start_timestamp)].index
        except TypeError:
            msg = ""
            self.Error(msg, exc_info=True)
        else:
            df = df.loc[strt_idx]
            end_idx = df[df["Sample Time"] <= end_ts].index
            df = df.loc[end_idx]
            # Check for gaps and warn
            time_stamps = df["Sample Time"].values
            ts_diff = np.diff(time_stamps)
            time_gaps = np.where(ts_diff > time_gap)
            for idx in time_gaps[0]:
                ts = time_stamps[idx]
                diff_time = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d_%H:%M")
                diff = ts_diff[idx]
                self.Step("Time gap of {}s found at {} in PDU samples.".format(diff, diff_time))
            # Convert power column to floats and build new array
            df_list = np.asarray(df.values.tolist())
            power_col = [x.split(",") for x in df_list[:, 3]]
            power_col = [[float(x) for x in y] for y in power_col]
            curr_col = [x.split(",") for x in df_list[:, 2]]
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
                start_time = datetime.fromtimestamp(rolled_up_samples[0][0]).strftime("%Y-%m-%d %H:%M:%S")
                end_time = datetime.fromtimestamp(rolled_up_samples[-1][0]).strftime("%Y-%m-%d %H:%M:%S")
                ru_smpls = np.asarray(rolled_up_samples)
                tot_power = ru_smpls[:, 1:4].sum(axis=1)
                self.Step("Compile Power consumption report while running SFDR test.")
                self.Progress("Power report from {} to {}".format(start_time, end_time))
                self.Progress("Average sample time: {}s".format(int(np.diff(ru_smpls[:, 0]).mean())))
                # Add samples for pdus in same rack
                rack_samples = {x[: x.find("-")]: [] for x in pdus}
                for name in pdu_samples:
                    rack_name = name[: name.find("-")]
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
                    self.Step("Measure CBF Power rack and confirm power consumption is less than 6.25kW")
                    msg = "Measured power for rack {} ({:.2f}kW) is less than {}kW".format(
                        rack, watts, max_power_per_rack
                    )
                    Aqf.less(watts, max_power_per_rack, msg)
                    phase = np.zeros(3)
                    for i, x in enumerate(phase):
                        phase[i] = curr[:, i].mean()
                    self.Step("Measure CBF Power and confirm power consumption is less than 60kW")
                    self.Progress(
                        "Average current per phase for rack {}: P1={:.2f}A, P2={:.2f}A, "
                        "P3={:.2f}A".format(rack, phase[0], phase[1], phase[2])
                    )
                    ph_m = np.max(phase)
                    max_diff = np.max([100 * (x / ph_m) for x in ph_m - phase])
                    max_diff = float("{:.1f}".format(max_diff))
                    self.Step("Measure CBF Peak Power and confirm power consumption is less than 60kW")
                    msg = "Maximum difference in current per phase for rack {} ({:.1f}%) is " "less than {}%".format(
                        rack, max_diff, max_power_diff_per_rack
                    )
                    # Aqf.less(max_diff,max_power_diff_per_rack,msg)
                    # Aqf.waived(msg)
                watts = tot_power.mean()
                msg = "Measured power for CBF ({:.2f}kW) is less than {}kW".format(watts, max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)
                watts = tot_power.max()
                msg = "Measured peak power for CBF ({:.2f}kW) is less than {}kW".format(watts, max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)


class TestTimeout:
    """
    Test Timeout class using ALARM signal.
    :param: seconds -> Int
    :param: error_message -> Str
    :rtype: None
    """

    class TestTimeoutError(Exception):
        """Custom TestTimeoutError exception"""

        pass

    def __init__(self, seconds=1, error_message="Test Timed-out"):
        self.seconds = seconds
        self.error_message = "".join([error_message, " after {} seconds".format(self.seconds)])

    def handle_timeout(self, signum, frame):
        raise TestTimeout.TestTimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class GetSensors(object):
    """Easily get sensor values without much work"""

    def __init__(self, corr_fix):
        self.req = corr_fix.katcp_rct.req
        self.sensors = corr_fix.katcp_rct.sensor

    def get_value(self, _name, exact=False):
        """
        Get sensor Value(s)

        Parameters
        ----------
        str: sensor name e.g. n_bls

        Return
        ---------
        List or Str or None: sensor value"""
        if exact:
            for s in dir(self.sensors):
                if s == _name:
                    return getattr(self.sensors, s).get_value()
        else:
            if any(_name in s for s in dir(self.sensors)):
                _attribute = [s for s in dir(self.sensors) if _name in s][0]
                return getattr(self.sensors, _attribute).get_value()

    @property
    def input_labels(self):
        """
        Simplified input labels(s)

        Return
        ---------
        List: simplified input labels
        """
        try:
            input_labelling = eval(self.sensors.input_labelling.get_value())
            input_labels = [x[0] for x in [list(i) for i in input_labelling]]
        except Exception:
            input_labels = str(self.req.input_labels()).split()[2:]
        return input_labels

    @property
    def input_labels_pols(self):
        """
        Input labels(s) including polarisation

        Return
        ---------
        List: simplified input labels, label polarisation
        """
        try:
            input_labelling = eval(self.sensors.input_labelling.get_value())
            input_labels = [(x[0],x[-1]) for x in [list(i) for i in input_labelling]]
        except Exception as e:
            Aqf.failed('Failed reading input labels: {}'.format(e))
        return input_labels

    @property
    def custom_input_labels(self):
        """
        Simplified custom input labels(s)

        Return
        ---------
        List: simplified custom input labels
        """
        n_ants = int(self.get_value("n_ants"))
        return ["inp{:03d}{}".format(x, i) for x in xrange(n_ants) for i in "xy"]

    @property
    def ch_center_freqs(self):
        """
        Calculates the center frequencies of all channels.
        First channel center frequency is 0 or start of narrow band band.
        Second element can be used as the channel bandwidth, for narrowband = [1]-[0]

        Return
        ---------
        List: channel center frequencies
        """
        center_f  = float(self.get_value("antenna_channelised_voltage_center_freq"))
        bandwidth = float(self.get_value("antenna_channelised_voltage_bandwidth"))
        n_chans   = float(self.get_value("antenna_channelised_voltage_n_chans"))
        ch_bandwidth = bandwidth / n_chans
        f_start = center_f - (bandwidth/2.) # Center freq of the first channel
        return f_start + np.arange(n_chans) * ch_bandwidth

    @property
    def sample_period(self):
        """
        Get sample rate and return sample period
        """
        return 1 / float(self.get_value("adc_sample_rate"))

    @property
    def fft_period(self):
        """
        Get FFT Period
        """
        return self.sample_period * 2 * float(self.get_value("antenna_channelised_voltage_n_chans"))

    @property
    def delta_f(self):
        """
        Get Correlator bandwidth
        """
        return float(self.get_value("antenna_channelised_voltage_bandwidth") 
                / (self.get_value("antenna_channelised_voltage_n_chans") - 1))

    def calc_freq_samples(self, dhost, chan, samples_per_chan, chans_around=0):
        """Calculate frequency points to sweep over a test channel.

        Parameters
        =========
        chan : int
           Channel number around which to place frequency samples
        samples_per_chan: int
           Number of frequency points per channel
        chans_around: int
           Number of channels to include around the test channel. I.e. value 1 will
           include one extra channel above and one below the test channel.

        Will put frequency sample on channel boundary if 2 or more points per channel are
        requested, and if will place a point in the centre of the channel if an odd number
        of points are specified.

        """
        
        assert samples_per_chan > 0
        assert chans_around > 0
        assert 0 <= chan < self.get_value("antenna_channelised_voltage_n_chans")
        assert 0 <= chan + chans_around < self.get_value("antenna_channelised_voltage_n_chans")
        assert 0 <= chan - chans_around < self.get_value("antenna_channelised_voltage_n_chans")

        start_chan = chan - chans_around
        end_chan = chan + chans_around
        if samples_per_chan == 1:
            return self.ch_center_freqs[start_chan: end_chan + 1]
        start_freq = self.ch_center_freqs[start_chan] - self.delta_f / 2
        end_freq = self.ch_center_freqs[end_chan] + self.delta_f / 2
        sample_spacing = self.delta_f / (samples_per_chan - 1)
        num_samples = int(np.round((end_freq - start_freq) / sample_spacing)) + 1
        
        #Find the dsim set step size
        start = self.get_value("antenna_channelised_voltage_center_freq")
        dhost.sine_sources.sin_0.set(frequency=start)
        curr_setf = dhost.sine_sources.sin_0.frequency
        for i in range(20):
            currf = start_freq+i
            dhost.sine_sources.sin_0.set(frequency=currf)
            setf = dhost.sine_sources.sin_0.frequency
            if setf != curr_setf:
                dsim_step = setf-curr_setf
                curr_setf = setf
        #Find actual center frequency
        req_samples = np.linspace(start_freq, end_freq, 
                num_samples, endpoint = False)
        req_step  = np.diff(req_samples)[0]
        cent_indx = int(num_samples/2)
        cent_freq = req_samples[cent_indx]
        real_step = dsim_step*round(req_step/dsim_step)
        dhost.sine_sources.sin_0.set(frequency=cent_freq)
        real_freq = dhost.sine_sources.sin_0.frequency
        first_half = np.asarray([real_freq-x*real_step 
                for x in range(cent_indx+1)])
        first_half = np.flip(first_half,0)
        second_half = np.asarray([real_freq+x*real_step 
                for x in range(1, num_samples-cent_indx)])
        return np.concatenate((first_half,second_half))

class CSV_Reader(object):
    """
    Manual Tests CSV reader

    Parameters
    ---------
        csv_filename: str, Valid path to csv file/url
        set_index: str, If you want to change the index, set name
    Returns
    -------
        result: Pandas DataFrame
    """

    def __init__(self, csv_filename, set_index=None):
        self.csv_filename = csv_filename
        self.set_index = set_index

    @property
    def load_csv(self):
        """
        Load csv file

        Parameters
        ----------
            object

        Returns
        -------
            result: Pandas DataFrame
        """
        try:
            assert self.csv_filename
            df = pd.read_csv(self.csv_filename)
            df = df.replace(np.nan, "TBD", regex=True)
            df = df.fillna(method="ffill")
        except Exception:
            LOGGER.error("could not load the csv file")
            return False
        else:
            return df.set_index(self.set_index) if self.set_index else df

    def csv_to_dict(self, ve_number=None):
        """
        CSV contents to Dict

        Parameters
        ----------
            ve_number: Verification Event Number e.g. CBF.V.1.11

        Returns
        -------
        result: dict
        """
        return dict(self.load_csv.loc[ve_number]) if ve_number else None


class AqfReporter(object):

    def Failed(self, msg, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        Aqf.failed(msg)
        self.logger.warn("-> Line:%d: - %s" % (caller.lineno, msg))

    def Passed(self, msg, *args, **kwargs):
        # caller = getframeinfo(stack()[1][0])
        Aqf.passed(msg)
        # self.logger.warn("-> Line:%d: - %s" % (caller.lineno, msg))

    def Error(self, msg, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        Aqf.failed(msg)
        exception_info = kwargs.get('exc_info', False)
        self.logger.error("-> Line:%d: - %s" % (caller.lineno, msg), exc_info=exception_info)

    def Step(self, msg, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        Aqf.step(msg)
        self.logger.debug("-> Line:%d: - %s" % (caller.lineno, msg))

    def Progress(self, msg, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        Aqf.progress(msg)
        self.logger.info("-> Line:%d: - %s" % (caller.lineno, msg))

    def Note(self, msg, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        Aqf.note(msg)
        self.logger.info("-> Line:%d: - %s" % (caller.lineno, msg))


@contextmanager
def RunTestWithTimeout(test_timeout, errmsg="Test Timed-out"):
    """
    Context manager to execute tests with a timeout
    :param: test_timeout : int
    :param: errmsg : str
    :rtype: None
    """
    try:
        with TestTimeout(seconds=test_timeout):
            yield
    except Exception:
        LOGGER.exception(errmsg)
        Aqf.failed(errmsg)
        Aqf.end(traceback=True)


@contextmanager
def ignored(*exceptions):
    """
    Context manager to ignore specified exceptions
    :param: Exception
    :rtype: None
    """
    try:
        yield
    except exceptions:
        pass


def FPGA_Connect(hosts, _timeout=30):
    """Utility to connect to hosts via Casperfpga"""
    fpgas = False
    retry = 10
    while not fpgas:
        try:
            fpgas = threaded_create_fpgas_from_hosts(hosts, timeout=_timeout, logger=LOGGER)
        except Exception:
            retry -= 1
            if retry == 0:
                LOGGER.error("ERROR: Could not connect to the SKARABs")
                return False
    return fpgas


def Report_Images(image_list, caption_list=[""]):
    """Add an image to the report

    Parameters
    ----------
    image_list : list
        list of names of the image files.
    caption : list
        List of caption text to go with the image

    Note a copy of the file will be made, and the test name as well as step number
    will be prepended to filename to ensure that it is unique
    """
    for image, caption in zip(image_list, caption_list):
        LOGGER.info("Adding image to report: %s" % image)
        Aqf.image(image, caption)



def array_release_x(f):
    """
    Custom decorator and flag for decorating tests that you would need a QTP/QTR generated automagically

    Usage CLI:
        Following command can be used in the command line to narrow the execution of the test to
        the ones marked with @array_release_x

        nosetests -a array_release_x

    Usage in Test:
        @array_release_x
        def test_channelistion(self):
            pass
    """
    return attr("array_release_x")(f)

def subset(f):
    """
    Custom decorator and flag for decorating tests that you would need a QTP/QTR generated automagically

    Usage CLI:
        Following command can be used in the command line to narrow the execution of the test to
        the ones marked with @subset

        nosetests -a subset

    Usage in Test:
        @subset
        def test_channelistion(self):
            pass
    """
    return attr("subset")(f)

def beamforming(f):
    """
    Custom decorator and flag for decorating tests that you would need a QTP/QTR generated automagically

    Usage CLI:
        Following command can be used in the command line to narrow the execution of the test to
        the ones marked with @subset

        nosetests -a beamforming

    Usage in Test:
        @beamforming
        def test_channelistion(self):
            pass
    """
    return attr("beamforming")(f)


def wipd(f):
    """
    - "work in progress decorator"
    Custom decorator and flag.

    # Then "nosetests -a wip" can be used at the command line to narrow the execution of the test to
    # the ones marked with @wipd

    Usage:
        @widp
        def test_channelistion(self):
            pass
    """
    return attr("wip")(f)


def flatten(lst):
    """
    The flatten function here turns the list into a string, takes out all of the square brackets,
    attaches square brackets back onto the ends, and turns it back into a list.
    """
    return eval('[' + str(lst).replace('[', '').replace(']', '') + ']')


def executed_by():
    """Get who ran the test."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
        if user == "root":
            raise OSError
        Aqf.hop(
            "Test run by: {} on {} system on {}.\n".format(
                user, os.uname()[1].upper(), time.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
    except Exception:
        LOGGER.error("Failed to determine who ran the test")
        Aqf.hop(
            "Test run by: Jenkins on system {} on {}.\n".format(os.uname()[1].upper(), time.ctime()))


def encode_passwd(pw_encrypt, key=None):
    """This function encrypts a string with base64 algorithm
    :param: pw_encrypt: Str

    :param: secret key : Str = 16 chars long
        Keep your secret_key safe!
    :rtype: encrypted password
    """
    if key is not None:
        _cipher = AES.new(key, AES.MODE_ECB)
        encoded = base64.b64encode(_cipher.encrypt(pw_encrypt.rjust(32)))
        return encoded


def decode_passwd(pw_decrypt, key=None):
    """This function decrypts a string with base64 algorithm
    :param: pw_decrypt: Str

    :param: secret key : Str = 16 chars long
        Keep your secret_key safe!
    :rtype: decrypted password
    """
    if key is not None:
        _cipher = AES.new(key, AES.MODE_ECB)
        decoded = _cipher.decrypt(base64.b64decode(pw_decrypt))
        return decoded.strip()


def ip2int(ipstr):
    return struct.unpack("!I", socket.inet_aton(ipstr))[0]


def int2ip(n):
    return socket.inet_ntoa(struct.pack("!I", n))


def complexise(input_data):
    """Convert input data shape (X,2) to complex shape (X)
    :param input_data: Xeng_Raw
    """
    return input_data[:, 0] + input_data[:, 1] * 1j


def magnetise(input_data):
    """Convert input data shape (X,2) to complex shape (X) and
       Calculate the absolute value element-wise.
       :param input_data: Xeng_Raw
    """
    id_c = complexise(input_data)
    id_m = np.abs(id_c)
    return id_m


def normalise(input_data):
    return input_data / VACC_FULL_RANGE


def normalised_magnitude(input_data):
    return normalise(magnetise(input_data))


def loggerise(data, dynamic_range=70, normalise=False, normalise_to=None, no_clip=False):
    with np.errstate(divide="ignore"):
        log_data = 10 * np.log10(data)
    if normalise_to:
        max_log = normalise_to
    else:
        max_log = np.max(log_data)
    if not (no_clip):
        min_log_clip = max_log - dynamic_range
        log_data[log_data < min_log_clip] = min_log_clip
    if normalise:
        log_data = np.asarray(log_data) - np.max(log_data)
    return log_data


def baseline_checker(xeng_raw, check_fn):
    """Apply a test function to correlator data one baseline at a time

    Returns a set of all the baseline indices for which the test matches
    :rtype: dict
    """
    baselines = set()
    for _baseline in range(xeng_raw.shape[1]):
        if check_fn(xeng_raw[:, _baseline, :]):
            baselines.add(_baseline)
    return baselines


def zero_baselines(xeng_raw):
    """Return baseline indices that have all-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.all(bldata == 0))


def nonzero_baselines(xeng_raw):
    """Return baseline indices that have some non-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.any(bldata != 0))


def all_nonzero_baselines(xeng_raw):
    """Return baseline indices that have all non-zero data"""

    def non_zerobls(bldata):
        return np.all(np.linalg.norm(bldata.astype(np.float64), axis=1) != 0)

    return baseline_checker(xeng_raw, non_zerobls)


def init_dsim_sources(dhost):
    """Select dsim signal output, zero all sources, output scaling to 1

    Also clear noise diode and adc over-range flags
    """
    # Reset flags
    LOGGER.info("Reset digitiser simulator to all Zeros")
    try:
        dhost.registers.flag_setup.write(adc_flag=0, ndiode_flag=0, load_flags="pulse")
    except Exception:
        LOGGER.error("Failed to set dhost flag registers.")
        pass

    try:
        for sin_source in dhost.sine_sources:
            sin_source.set(frequency=0, scale=0)
            assert sin_source.frequency == sin_source.scale == 0
            try:
                if sin_source.name != "corr":
                    sin_source.set(repeat_n=0)
                    assert sin_source.repeat == 0
            except Exception:
                LOGGER.exception("Failed to reset repeat on sin_%s" % sin_source.name)
            LOGGER.debug("Digitiser simulator cw source %s reset to Zeros" % sin_source.name)
    except Exception:
        LOGGER.exception("Failed to reset sine sources on dhost.")

    try:
        for noise_source in dhost.noise_sources:
            noise_source.set(scale=0)
            assert noise_source.scale == 0
            LOGGER.debug("Digitiser simulator awg sources %s reset to Zeros" % noise_source.name)
    except Exception:
        LOGGER.error("Failed to reset noise sources on dhost.")

    try:
        for output in dhost.outputs:
            output.select_output("signal")
            output.scale_output(1)
            LOGGER.debug("Digitiser simulator signal output %s selected." % output.name)
    except Exception:
        LOGGER.error("Failed to select output dhost.")


def get_dsim_source_info(dsim):
    """Return a dict with all the current sine, noise and output settings of a dsim"""
    info = dict(sin_sources={}, noise_sources={}, outputs={})
    for sin_src in dsim.sine_sources:
        info["sin_sources"][sin_src.name] = dict(scale=sin_src.scale, frequency=sin_src.frequency)
    for noise_src in dsim.noise_sources:
        info["noise_sources"][noise_src.name] = dict(scale=noise_src.scale)

    for output in dsim.outputs:
        info["outputs"][output.name] = dict(output_type=output.output_type, scale=output.scale)
    return info


def iterate_recursive_dict(dictionary, keys=()):
    """Generator; walk through a recursive dict structure

    yields the compound key and value of each leaf.

    Example
    =======
    ::
      eg = {
      'key_1': 'value_1',
      'key_2': {'key_21': 21,
              'key_22': {'key_221': 221, 'key_222': 222}}}

      for compound_key, val in iterate_recursive_dict(eg):
         print '{}: {}'.format(compound_key, val)

    should produce output:

    ::
      ('key_1', ): value_1
      ('key_2', 'key_21'): 21
      ('key_2', 'key_22', 'key_221'): 221
      ('key_2', 'key_22', 'key_222'): 222

    """
    if isinstance(dictionary, Mapping):
        for k in dictionary:
            for rv in iterate_recursive_dict(dictionary[k], keys + (k,)):
                yield rv
    else:
        yield (keys, dictionary)


def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0,
       figure out if VACC is rooted by 1"""
    input0 = np.abs(complexise(xeng_raw[:, 0]))
    input1 = np.abs(complexise(xeng_raw[:, 1]))
    if np.max(input0) > float(0) and np.max(input1) == float(0):
        # We expect autocorr in baseline 0 to be nonzero if the vacc is
        # properly aligned, hence no offset
        return 0
    elif np.max(input1) > float(0) and np.max(input0) == float(0):
        return 1
    else:
        return False


def get_bit_flag(packed, flag_bit):
    flag_mask = 1 << flag_bit
    flag = bool(packed & flag_mask)
    return flag


def get_set_bits(packed, consider_bits=None):
    packed = int(packed)
    set_bits = set()
    for bit in range(packed.bit_length()):
        if get_bit_flag(packed, bit):
            set_bits.add(bit)
    if consider_bits is not None:
        set_bits = set_bits.intersection(consider_bits)
    return set_bits


def get_pfb_counts(status_dict):
    """Checks if FFT overflows and QDR status on roaches
    Param: fhosts items
    Return: Dict:
        Dictionary with pfb counts
    """
    pfb_list = {}
    for host, pfb_value in status_dict:
        pfb_list[host] = (pfb_value["pfb_of0_cnt"], pfb_value["pfb_of1_cnt"])
    return pfb_list


def get_delay_bounds(correlator):
    """
    Parameters
    ----------
    correlator - As displayed in you on board flight manual

    Returns
    -------
    Dictionary containing minimum and maximum values for delay, delay rate,
    phase offset and phase offset rate

    """
    fhost = correlator.fhosts[0]
    # Get maximum delay value
    reg_info = fhost.registers.delay0.block_info
    reg_bw = int(reg_info["bitwidths"])
    reg_bp = int(reg_info["bin_pts"])
    max_delay = 2 ** (reg_bw - reg_bp) - 1 / float(2 ** reg_bp)
    max_delay = max_delay / correlator.sample_rate_hz
    min_delay = 0
    # Get maximum delay rate value
    reg_info = fhost.registers.delta_delay0.block_info
    _b = int(reg_info["bin_pts"])
    max_positive_delta_delay = 1 - 1 / float(2 ** _b)
    max_negative_delta_delay = -1 + 1 / float(2 ** _b)
    # Get max/min phase offset
    reg_info = fhost.registers.phase0.block_info
    b_str = reg_info["bin_pts"]
    _b = int(b_str[1: len(b_str) - 1].rsplit(" ")[0])
    max_positive_phase_offset = 1 - 1 / float(2 ** _b)
    max_negative_phase_offset = -1 + 1 / float(2 ** _b)
    max_positive_phase_offset *= float(np.pi)
    max_negative_phase_offset *= float(np.pi)
    # Get max/min phase rate
    b_str = reg_info["bin_pts"]
    _b = int(b_str[1: len(b_str) - 1].rsplit(" ")[1])
    max_positive_delta_phase = 1 - 1 / float(2 ** _b)
    max_negative_delta_phase = -1 + 1 / float(2 ** _b)
    # As per fhost_fpga
    bitshift_schedule = 23
    bitshift = 2 ** bitshift_schedule
    max_positive_delta_phase = (
        max_positive_delta_phase * float(np.pi) * correlator.sample_rate_hz
    ) / bitshift
    max_negative_delta_phase = (
        max_negative_delta_phase * float(np.pi) * correlator.sample_rate_hz
    ) / bitshift
    return {
        "max_delay_value": max_delay,
        "min_delay_value": min_delay,
        "max_delay_rate": max_positive_delta_delay,
        "min_delay_rate": max_negative_delta_delay,
        "max_phase_offset": max_positive_phase_offset,
        "min_phase_offset": max_negative_phase_offset,
        "max_phase_rate": max_positive_delta_phase,
        "min_phase_rate": max_negative_delta_phase,
    }


def disable_warnings_messages():
    """This function disables all error warning messages
    """
    import matplotlib

    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    # Ignoring all warnings raised when casting a complex dtype to a real dtype.
    warnings.simplefilter("ignore", np.ComplexWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    ignored_loggers = [
        "casperfpga",
        "casperfpga.casperfpga",
        "casperfpga.bitfield",
        "casperfpga.katcp_fpg",
        "casperfpga.memory",
        "casperfpga.register",
        "casperfpga.transport_katcp",
        "casperfpga.transort_skarab",
        "corr2.corr_rx",
        "corr2.fhost_fpga",
        "corr2.fhost_fpga",
        "corr2.fxcorrelator_fengops",
        "corr2.xhost_fpga",
        "katcp",
        "spead2",
        "tornado.application",
        "corr2",
    ]
    # Ignore all loggings except Critical if any
    for logger_name in ignored_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger("nose.plugins.nosekatreport").setLevel(logging.INFO)


def retryloop(attempts, timeout):
    """
    Usage:

    for retry in retryloop(10, timeout=30):
        try:
            something
        except SomeException:
            retry()

    for retry in retryloop(10, timeout=30):
        something
        if somecondition:
            retry()

    """
    starttime = time.time()
    success = set()
    for i in range(attempts):
        success.add(True)
        yield success.clear
        if success:
            return
        if starttime + timeout < time.time():
            break
    raise RetryError("Failed to after %s attempts" % attempts)


def human_readable_ip(hex_ip):
    hex_ip = hex_ip[2:]
    return ".".join([str(int(x + y, 16)) for x, y in zip(hex_ip[::2], hex_ip[1::2])])
